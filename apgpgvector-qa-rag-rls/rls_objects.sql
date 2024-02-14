CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS public.qa_rag_rls;

CREATE TABLE IF NOT EXISTS public.qa_rag_rls
(
    username text,
    document text,
    embedding vector(1536)
);

CREATE INDEX ON public.qa_rag_rls
    USING hnsw (embedding vector_cosine_ops)
    WITH  (m = 16, ef_construction = 64);


CREATE POLICY qa_rag_rls_rls_policy ON public.qa_rag_rls FOR ALL TO PUBLIC USING (username=current_user or username='generic');

ALTER TABLE public.qa_rag_rls ENABLE ROW LEVEL SECURITY;

DROP ROLE IF EXISTS user1;
CREATE ROLE user1 login ;
DROP ROLE IF EXISTS user2;
CREATE ROLE user2 login ;
DROP ROLE IF EXISTS user3;
CREATE ROLE user3 login ;
DROP ROLE IF EXISTS generic;
CREATE ROLE generic login ;

grant select on public.qa_rag_rls to user1;
grant select on public.qa_rag_rls to user2;
grant select on public.qa_rag_rls to user3;
grant select on public.qa_rag_rls to generic;
