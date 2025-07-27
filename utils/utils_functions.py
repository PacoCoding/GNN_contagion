def make_preprocessor():
    return Pipeline([
        ("clip_equity", FunctionTransformer(
            lambda X: np.column_stack([X[:,0], X[:,1], np.clip(X[:,2],0,None), X[:,3], X[:,4], X[:,5:]]), validate=True)),
        ("clip_degrees", FunctionTransformer(
            lambda X: np.column_stack([X[:,:5], np.clip(X[:,5],0,None), np.clip(X[:,6],0,None), np.clip(X[:,7],0,None), np.clip(X[:,8],0,None)]), validate=True)),
        ("log1p", FunctionTransformer(np.log1p, validate=True)),
        ("robust", RobustScaler()),
    ])

def enrich_nodes(nodes_df, edges_df):
    df = nodes_df.copy()
    if 'index' not in df.columns:
        raise KeyError("'index' column missing")
    df.insert(0, 'BankID', df.index.astype(str))
    e = edges_df.copy()
    e['Sourceid'] = e['Sourceid'].astype(str).str.strip()
    e['Targetid'] = e['Targetid'].astype(str).str.strip()
    in_deg   = e.groupby('Targetid').size().rename('in_degree')
    out_deg  = e.groupby('Sourceid').size().rename('out_degree')
    in_wdeg  = e.groupby('Targetid')['Weights'].sum().rename('in_wdeg')
    out_wdeg = e.groupby('Sourceid')['Weights'].sum().rename('out_wdeg')
    node_feats = (pd.concat([in_deg, out_deg, in_wdeg, out_wdeg], axis=1)
                  .fillna(0)
                  .reset_index()
                  .rename(columns={'index':'BankID'}))
    merged = pd.merge(df, node_feats, on='BankID', how='left')
    merged[['in_degree','out_degree','in_wdeg','out_wdeg']] = merged[['in_degree','out_degree','in_wdeg','out_wdeg']].fillna(0)
    if TARGET in df.columns:
        merged[TARGET] = df[TARGET].values
    return merged
