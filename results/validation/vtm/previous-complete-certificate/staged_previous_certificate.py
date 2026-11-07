"""Private #182 v2: fail early, while retaining every publication proof gate.

Only the speculative whole-previous-mesh path uses this schedule. Ordinary
construction and fragment completion retain their existing certificates.
"""
import warp as wp
import warpower._delaunay_certificate as dc
from whole_previous_certificate import classify_whole
from warpower._reuse_delaunay import ReuseDelaunayWorkspace, completed


def check_previous(c, tree):
    c.error.zero_()
    c.statistics.zero_()
    c._used.zero_()
    c._boundary_count.zero_()
    if c._anchors is not None:
        c._anchors.zero_()
        c._local.zero_()
        c._global.zero_()
    c._launch(dc.select_positive, len(c.input_tets), [c.points, c.input_tets,
        c.input_count, tree.owners, c._selected, c.error])
    c._scan.run(c._selected, c._prefix)
    c._launch(dc.compact, len(c.input_tets), [c.input_tets, c._selected,
        c._prefix, tree.owners, c.tetrahedra, c.tetrahedra_count, c._used])
    # Canonical coverage is necessary on either the local or global route.
    c._launch(dc.coverage, len(c.points), [tree.owners, c._used,
        c.tetrahedra_count, c.error])

    def only_if_valid(fn):
        c._launch(dc.ready, 1, [c.error, c._ready])
        wp.capture_if(c._ready, fn)

    def topology():
        c._table.fill_(-1)
        c._occurrences.zero_()
        c._balance.zero_()
        c._launch(dc.identity, len(c._parents), [c._parents])
        c._launch(dc.index_faces, 4 * len(c.tetrahedra), [c.tetrahedra,
            c.tetrahedra_count, c._table, c._occurrences, c._balance, c._parents])
        c._launch(dc.collect_boundary_faces, len(c._table), [c._table,
            c._occurrences, c._balance, c._boundary, c._boundary_count, c.error])

    def necessary_spheres():
        if c._anchors is None:
            c._global_spheres(tree)
        else:
            # Every paired-face query is also one of the global all-site
            # queries. Failure conclusively rules out a global certificate;
            # success alone does NOT authorize publication without hull proof.
            c._launch(dc.check_local_spheres, 4 * len(c.tetrahedra), [c.points,
                c.tetrahedra, c.tetrahedra_count, c._table, c.error, c.statistics])

    def select_proof():
        c._launch(dc.find_anchors, len(c._anchors) * (1 << c._face_depth),
            [c.points, c.tetrahedra, c._boundary, c._boundary_count, tree,
             c._face_depth, c._anchors])
        c._launch(dc.select_certificate, 1, [c._anchors, c._boundary_count,
            c.error, c._local, c._global])
        # The local checks have run already. Without a strict anchor, retain
        # the full global check: paired faces alone cannot exclude double covers.
        wp.capture_if(c._global, lambda: c._global_spheres(tree))

    only_if_valid(topology)
    only_if_valid(necessary_spheres)
    only_if_valid(lambda: c._hull_support(tree))
    if c._anchors is not None:
        only_if_valid(select_proof)


instances = []


def install():
    original_init = ReuseDelaunayWorkspace.__init__
    original_record = ReuseDelaunayWorkspace._record_reuse
    original_attempt = ReuseDelaunayWorkspace._attempt_reuse

    def init(self, *args, **kwargs):
        self._whole_enabled = False
        original_init(self, *args, **kwargs)
        with wp.ScopedDevice(self.device):
            self._whole_status = wp.zeros(3, dtype=int)
            self._whole_accepted = self._whole_status[1:2]
            wp.load_module(module=classify_whole.module, device=self.device)
            self._whole_enabled = True
            with wp.ScopedCapture() as capture:
                self._record_reuse()
            self._reuse_graph = capture.graph
        wp.synchronize_device(self.device)
        instances.append(self)

    def record(self):
        if self._whole_enabled:
            self._whole_status.zero_()
        original_record(self)

    def publish(self):
        wp.copy(self._candidate, self.tetrahedra)
        wp.copy(self._candidate_count, self.tetrahedra_count)
        self._reuse_completion._certificate.proof_statistics.zero_()

    def fragment(self):
        self._reuse_completion.run()
        self._launch(completed, 1, [self._reuse_filter.error,
            self._reuse_completion.error, self._reuse_completion.tetrahedra_count,
            len(self.tetrahedra), self.reuse_error, self.reuse_accepted])
        wp.capture_if(self.reuse_accepted, self._publish_reuse)

    def attempt(self):
        if not self._whole_enabled:
            return original_attempt(self)
        self._refresh_original()
        wp.copy(self._candidate, self._previous)
        wp.copy(self._candidate_count, self._previous_count)
        check_previous(self._certificate, self._original.tree)
        self._launch(classify_whole, 1, [self.certificate_error,
            self.tetrahedra_count, len(self.tetrahedra), self._whole_status,
            self.reuse_accepted, self.reuse_error, self.error, self.recovered,
            self._reuse_ready])
        wp.capture_if(self._whole_accepted, lambda: publish(self))
        wp.capture_if(self._reuse_ready, lambda: fragment(self))

    ReuseDelaunayWorkspace.__init__ = init
    ReuseDelaunayWorkspace._record_reuse = record
    ReuseDelaunayWorkspace._attempt_reuse = attempt
