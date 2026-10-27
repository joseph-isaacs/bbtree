//! A map's page arena (experiment: arena edges): one reserved block, of a
//! capped size and aligned to it, that every internal, leaf and root page
//! of the map comes from, so that an edge can be a page number within it
//! (`tree/node/page.rs`, `EdgeFormat`) and the block's start is any page's
//! address with its low `SIZE_BITS` cleared.
//!
//! The block is reserved as twice the cap, zeroed, from the global
//! allocator: on Linux that maps fresh pages that hold memory only once
//! touched. Pages bump from its aligned middle, and come back on a free
//! list per page size, zeroed again when reused. Heap pages, and anything
//! else, come from [`Global`].

use alloc::alloc::{alloc_zeroed, dealloc};
use alloc::rc::Rc;
use core::alloc::Layout;
use core::cell::Cell;
use core::ptr::{self, NonNull};

use crate::allocator::{AllocError, Allocator, Global};

/// A map's page arena (experiment: arena edges): pages of up to
/// [`Self::SIZE`] bytes in all, for the internal nodes, leaves and root page
/// of the maps that share it, which clones of it do. For one thread.
#[derive(Clone)]
pub struct PageArena {
    block: Rc<ArenaBlock>,
}

/// The reserved block, and what of it is handed out.
struct ArenaBlock {
    /// The reservation, as the global allocator gave it.
    reservation: NonNull<u8>,
    /// Where the aligned, usable `PageArena::SIZE` bytes start.
    start: usize,
    /// How many bytes from `start` have been handed out, the first page
    /// left out so that none starts the block.
    bumped: Cell<usize>,
    /// Freed pages, by size: each a list linked through the pages' first
    /// word, or empty (size 0).
    free_lists: [FreeList; 4],
}

/// Freed pages of one size.
struct FreeList {
    page_size: Cell<usize>,
    head: Cell<*mut u8>,
}

impl PageArena {
    /// The cap, as a power of two: 4 GiB, so that 512-byte page numbers fill
    /// 23 bits, and with a leaf tag a 3-byte edge.
    pub const SIZE_BITS: u32 = 32;

    /// The cap on the bytes of pages the arena hands out.
    pub const SIZE: usize = 1 << Self::SIZE_BITS;

    /// The bytes the block leaves out at its start: no page starts the
    /// block, so no edge is 0, a zeroed slot.
    const RESERVED_START: usize = 4096;

    /// A new arena, its block reserved.
    ///
    /// # Panics
    ///
    /// If the reservation fails.
    pub fn new() -> Self {
        let layout = Self::reservation_layout();
        // SAFETY: the layout has a non-zero size.
        let reservation = NonNull::new(unsafe { alloc_zeroed(layout) })
            .expect("the page arena's reservation (experiment) failed");
        let start = reservation.addr().get().next_multiple_of(Self::SIZE);
        let free_list = || FreeList {
            page_size: Cell::new(0),
            head: Cell::new(ptr::null_mut()),
        };
        Self {
            block: Rc::new(ArenaBlock {
                reservation,
                start,
                bumped: Cell::new(Self::RESERVED_START),
                free_lists: [free_list(), free_list(), free_list(), free_list()],
            }),
        }
    }

    /// Twice the cap, so that an aligned cap fits in it, at an alignment the
    /// global allocator zeroes by mapping fresh pages, not by writing them.
    fn reservation_layout() -> Layout {
        Layout::from_size_align(2 * Self::SIZE, 16).expect("the reservation's layout")
    }

    /// The free list for pages of `page_size` bytes, which it takes over
    /// if no list has that size yet.
    fn free_list(&self, page_size: usize) -> &FreeList {
        let lists = &self.block.free_lists;
        lists
            .iter()
            .find(|list| list.page_size.get() == page_size)
            .or_else(|| lists.iter().find(|list| list.page_size.get() == 0))
            .inspect(|list| list.page_size.set(page_size))
            .expect("a map takes pages of at most four sizes")
    }

    /// A zeroed page for `layout`, from its free list, else bumped.
    fn allocate_page(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        let list = self.free_list(layout.size());
        if let Some(page) = NonNull::new(list.head.get()) {
            // SAFETY: a freed page of this size, in the block, whose first
            // word links the next one; it is handed out again, zeroed.
            unsafe {
                list.head.set(page.cast::<*mut u8>().read());
                page.write_bytes(0, layout.size());
            }
            return Ok(page);
        }
        let offset = self.block.bumped.get().next_multiple_of(layout.align());
        let end = offset.checked_add(layout.size()).ok_or(AllocError)?;
        if end > Self::SIZE {
            return Err(AllocError);
        }
        self.block.bumped.set(end);
        let page = (self.block.reservation.as_ptr()).with_addr(self.block.start + offset);
        // SAFETY: within the reservation, which is not null.
        Ok(unsafe { NonNull::new_unchecked(page) })
    }

    /// Takes back `page`, of `layout`, onto its free list.
    ///
    /// # Safety
    ///
    /// `page` came from [`Self::allocate_page`] for `layout`, and is given
    /// back once.
    unsafe fn deallocate_page(&self, page: NonNull<u8>, layout: Layout) {
        let list = self.free_list(layout.size());
        // SAFETY: the page is the caller's to give back, and holds a word.
        unsafe { page.cast::<*mut u8>().write(list.head.get()) };
        list.head.set(page.as_ptr());
    }
}

impl Default for PageArena {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ArenaBlock {
    fn drop(&mut self) {
        // SAFETY: the reservation came from `alloc_zeroed` with this layout,
        // and no page of it is used once every arena sharing it is gone.
        unsafe { dealloc(self.reservation.as_ptr(), PageArena::reservation_layout()) }
    }
}

// SAFETY: node pages come from the block: zeroed, aligned, apart, and valid
// until given back or until every clone, sharing the block, is dropped.
// Everything else is `Global`'s. Nothing unwinds but the `expect` on a map
// taking pages of a fifth size, which aborts the experiment.
unsafe impl Allocator for PageArena {
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        Global.allocate_zeroed(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded.
        unsafe { Global.deallocate(ptr, layout) }
    }

    fn allocate_internal(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.allocate_page(layout)
    }

    unsafe fn deallocate_internal(&self, page: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded.
        unsafe { self.deallocate_page(page, layout) }
    }

    fn allocate_leaf(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.allocate_page(layout)
    }

    unsafe fn deallocate_leaf(&self, page: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded.
        unsafe { self.deallocate_page(page, layout) }
    }

    fn allocate_root(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        self.allocate_page(layout)
    }

    unsafe fn deallocate_root(&self, page: NonNull<u8>, layout: Layout) {
        // SAFETY: forwarded.
        unsafe { self.deallocate_page(page, layout) }
    }
}
